?($	(?X???}ME?T?????e?c]??!46<?R??$	?2?@I@?(/U??@9?߅??!L?J#?9@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?x?&1??+??ݓ???A?X????Y?(\?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(????O??n??A?Ǻ????Y)\???(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?[ A????q?????A? ?rh???Y?+e?X??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)???????H.??A?W?2??Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=
??;M?O??AO??e?c??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?R???|гY???Aףp=
???Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?J?4??`??"????A?5?;N???Y ?o_Ι?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ܵ?|???7?A`????A??v????Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??^)??2U0*???AvOjM??YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??y?):???lV}???A?ͪ??V??YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?鷯??O??e?c??A?e??a???YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V?/?'??ףp=
???A??Q???Y^K?=???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&,Ԛ???????B?i??Ao??ʡ??Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ͪ????????<,??A|a2U0??YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?*?????f??j+??AEGr????Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?2ı.n???v??/??A8??d?`??YDio??ɔ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_)?Ǻ??9EGr???A????߾??Y?{??Pk??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&bX9????Y?8??m??A|??Pk???Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?K7?A`???Zd;??A\ A?c???Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~j?t????Zd;???AjM????Y??q????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??e?c]??Ԛ?????A??ǘ????Y??H?}??*	hffff"?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(~??k	??!J?!???B@)????_v??1?Σ?߂A@:Preprocessing2F
Iterator::ModelA??ǘ???!6??r??@@)??	h"l??152?<V?5@:Preprocessing2U
Iterator::Model::ParallelMapV2?_vO??!n R?x'@)?_vO??1n R?x'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?{??Pk??!e??Ɩ?P@)0?'???1?Dn$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???B?i??!ϋ??"]@)???B?i??1ϋ??"]@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?G?z??!*\?{+@):??H???1?,?M??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??????!p?b?182@)f??a?֤?1l??п@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?(??0??!<???;?@)?(??0??1<???;?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Nr?2?%@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	ؘ`?t??z`??MZ??+??ݓ???!?|гY???	!       "	!       *	!       2$	8`??????չ?6O????ǘ????!|??Pk???:	!       B	!       J$	??鯁????
̊ې??? ?rh??!?(\?????R	!       Z$	??鯁????
̊ې??? ?rh??!?(\?????JCPU_ONLYYNr?2?%@b Y      Y@q?????8U@"?
both?Your program is POTENTIALLY input-bound because 53.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?84.8875% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 