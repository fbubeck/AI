?($	?-Ϛs???NZ?39??_?L?J??!ı.n???$	?9?C?@?k??*F@?ا??X@!~*????3@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?b?=y???????ױ?A?8EGr???YC??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{?G?z??=?U?????A??y?)??Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/?$????|?5^??Ae?X???Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??N@a??=?U????AC?i?q???Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ı.n?????H.???A?%䃞???Yݵ?|г??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?):?????Pk?w???AjM??S??Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&NbX9???w-!?l??AL7?A`???Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??HP???V-??A%??C???Y2??%䃞?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&xz?,C??L?
F%u??A0L?
F%??YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	~??k	?????/?$??AM?O????Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??k	???????T????A?=yX???Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????????B?i??A?rh??|??Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???(???	??g????A?J?4??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&1???$???????A??&S??Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????Y?8??m??A?q?????Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????B??? c?ZB>??A=?U????Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?O??n???I+???A\???(\??Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6?;Nё??F%u???A0?'???Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ޓ??Z???|a2U0??A?>W[????Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<???x??#????A?A?f????YM?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&_?L?J?????3???A????Y2U0*???*	33333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????9#??!|%n?BA@)*??D???1??@??@@:Preprocessing2F
Iterator::ModelE???JY??!)+ivA@)6<?R?!??1XZ???5@:Preprocessing2U
Iterator::Model::ParallelMapV2??H.?!??!??-OŹ+@)??H.?!??1??-OŹ+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?0?*??!wk?s?DP@)??V?/???1??q??&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?6?[ ??!8??O?L @)?6?[ ??18??O?L @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??0?*??![6!g?.@)???B?i??1?M1???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[????<??!??|?dB3@)???H??1? O??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*M?O???!5?J???@)M?O???15?J???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9x>?Y?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	{HL????Xn?Bi??????3???!?I+???	!       "	!       *	!       2$	??B_?q??)??볱?????!?%䃞???:	!       B	!       J$	?|̜EC??8-ˉ??<,Ԛ???!C??6??R	!       Z$	?|̜EC??8-ˉ??<,Ԛ???!C??6??JCPU_ONLYYx>?Y?@b Y      Y@qWZ???A@"?
both?Your program is POTENTIALLY input-bound because 53.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?35.262% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 