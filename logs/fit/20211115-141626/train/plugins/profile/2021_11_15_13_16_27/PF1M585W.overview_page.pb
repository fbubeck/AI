?($	?wY???????????????߾??!?:pΈ???$	????K%@?g&?@?????@!??????3@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$M?O???j?t???A??ʡE??Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&z6?>W??o???T???AK?=?U??YO??e?c??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:pΈ?????ڊ?e??AǺ?????Y??D????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ףp=
???A??ǘ???AK?=?U??Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6?>W[???%??C???Am???{???Y???_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'???????+??????A?????B??YDio??ɤ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& o?ŏ???ׁsF???A?l??????Y????????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?S㥛???d?]K???Ao???T???Y??(\?¥?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?c?ZB?????_vO??A?46<??Y??_?L??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	M?J???}?5^?I??A??j+????Y7?[ A??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?HP???O@a????A???T????Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???JY????!?uq??Au????YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_?Q???=?U?????A???K7???Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&? ?	???Nё\?C??AOjM???Y??A?f??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??b?=???/L?
F??A?1??%???Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&%u????s?????A?8??m4??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???z6???c?]K???A
h"lxz??Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??B?i????&1???A8gDio??YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?sF??????????A??+e???YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'1?Z??????z??Ad?]K???Y???&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????߾????S㥛??A??y?):??Ye?X???*	    ??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatꕲq???!?e@
}VA@)a??+e??1??s???@:Preprocessing2F
Iterator::ModelM??St$??!???w??@@)?V?/?'??1n????g4@:Preprocessing2U
Iterator::Model::ParallelMapV2?c?ZB??!??????*@)?c?ZB??1??????*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?s?????!??D??P@)d]?Fx??1??<?"5(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceU???N@??!??<1? @)U???N@??1??<1? @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatea??+e??!1J??.@)????13ݔ?{c@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap%u???!??g?){3@)HP?sע?1S?
?Ț@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?(??0??!???3@)?(??0??1???3@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??0t	P@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	f}??????R?p?=????S㥛??!??ڊ?e??	!       "	!       *	!       2$	??Á???Z??)r?????y?):??!Ǻ?????:	!       B	!       J$	rĈ쭯???`Hz??e?X???!vq?-??R	!       Z$	rĈ쭯???`Hz??e?X???!vq?-??JCPU_ONLYY??0t	P@b Y      Y@qvm??F@"?
both?Your program is POTENTIALLY input-bound because 52.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?44.0231% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 